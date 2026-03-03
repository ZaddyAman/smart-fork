"""CLI for SmartFork."""

import sys
# CRITICAL: Configure UTF-8 BEFORE any Rich imports on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta
from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from loguru import logger
import sys
import json
import time

from .config import get_config, CONFIG_FILE
from .database.chroma_db import ChromaDatabase
from .indexer.indexer import FullIndexer, IncrementalIndexer
from .indexer.watcher import TranscriptWatcher
from .search.hybrid import HybridSearchEngine
from .fork.generator import ForkMDGenerator
from .intelligence.pre_compaction import CompactionManager
from .intelligence.clustering import SessionClusterer
from .intelligence.branching import BranchingTree
from .intelligence.privacy import PrivacyVault
from .intelligence.titling import TitleManager, TitleGenerator
from .indexer.parser import KiloCodeParser
from .ui.progress import (
    SmartForkProgress, display_discovery_phase,
    display_completion_summary, THEMES, DEFAULT_THEME,
)
from .ui.contextual_help import ContextualHelpManager, UserAction, get_help_manager
from .ui.interactive import start_interactive_shell

app = typer.Typer(help="SmartFork - AI Session Intelligence for Kilo Code")
console = Console()

# Initialize contextual help manager
help_manager = get_help_manager(console)


def setup_logging(log_level: str, log_file: Optional[Path] = None):
    """Setup logging configuration."""
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File logging
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="1 week"
        )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log file path")
):
    """SmartFork - AI Session Intelligence for Kilo Code."""
    config = get_config()
    log_level = "DEBUG" if verbose else config.log_level
    setup_logging(log_level, log_file)


@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch after indexing"),
):
    """Index all Kilo Code sessions."""
    config     = get_config()
    theme_name = getattr(config, "theme", DEFAULT_THEME)
    theme      = THEMES.get(theme_name, THEMES[DEFAULT_THEME])

    # Header panel
    path_str = str(config.kilo_code_tasks_path)
    if len(path_str) > 58:
        path_str = "…" + path_str[-55:]
    hdr = Text()
    hdr.append("⚡ SmartFork ", style=f"bold {theme['bars'][0]['color']}")
    hdr.append("Indexer\n",    style=f"bold {theme['text_primary']}")
    hdr.append(f"  {path_str}", style=f"dim {theme['text_muted']}")
    console.print(Panel(hdr, border_style=theme["panel_border"], box=box.ROUNDED, padding=(0, 2)))
    console.print()

    if not config.kilo_code_tasks_path.exists():
        console.print(f"[red]Tasks path not found:[/red] {config.kilo_code_tasks_path}")
        raise typer.Exit(1)

    db = ChromaDatabase(config.chroma_db_path)
    if force:
        console.print(f"[{theme['text_muted']}]Resetting database...[/{theme['text_muted']}]")
        db.reset()

    db_session_ids = set()
    try:
        db_session_ids = set(db.get_unique_sessions())
    except Exception:
        pass

    # Discovery
    all_sessions, new_count, _ = display_discovery_phase(
        tasks_path=config.kilo_code_tasks_path,
        db_session_ids=db_session_ids,
        console=console, theme_name=theme_name,
    )

    if not all_sessions:
        console.print(f"[{theme['text_muted']}]No sessions found.[/{theme['text_muted']}]")
        raise typer.Exit(0)

    sessions_to_index = [s for s in all_sessions if s.name not in db_session_ids]

    if not sessions_to_index:
        console.print(f"[{theme['done_color']}]✓[/{theme['done_color']}] All sessions already indexed.\n")
        if watch:
            _start_watch_mode(config, db, console, theme)
        raise typer.Exit(0)

    console.print(
        f"  [{theme['bars'][1]['color']}]→[/{theme['bars'][1]['color']}] "
        f"Indexing {len(sessions_to_index)} sessions...\n"
    )

    indexer    = FullIndexer(db, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    final_stats = None

    with SmartForkProgress(
        total_sessions=len(sessions_to_index),
        theme_name=theme_name,
        console=console,
    ) as prog:

        for i, session_dir in enumerate(sessions_to_index):
            sid = session_dir.name
            prog.set_session(sid)

            try:
                prog.set_phase("Parsing", 0.3)
                prog.set_phase("Embedding", 0.0)
                chunks = indexer.index_session(session_dir)
                prog.set_phase("Embedding", 1.0)

                # Try to get LLM-generated title
                title = ""
                try:
                    sc = db.get_session_chunks(sid)
                    if sc:
                        title = sc[0].metadata.session_title or ""
                except Exception:
                    pass
                if title:
                    prog.set_session(sid, title=title)

                prog.add_chunks(chunks)
                prog.advance()

            except Exception as e:
                logger.error(f"Failed to index {sid}: {e}")
                prog.add_error()
                prog.advance()

            prog.set_bm25((i+1) / len(sessions_to_index))

        prog.finish()
        final_stats = prog._stats

    # Completion
    total_db = 0
    try:
        total_db = len(db.get_unique_sessions())
    except Exception:
        pass

    if final_stats:
        display_completion_summary(
            stats=final_stats, total_db_sessions=total_db,
            console=console, theme_name=theme_name,
        )

    help_manager.show_after_command(
        UserAction.INDEX,
        db_session_count=total_db,
        processed=final_stats.indexed_sessions if final_stats else 0,
        failed=final_stats.errors if final_stats else 0,
    )

    if watch:
        _start_watch_mode(config, db, console, theme)


def _start_watch_mode(config, db, console, theme):
    """Helper to start watch mode after indexing."""
    console.print(
        f"\n  [{theme['bars'][0]['color']}]◉[/{theme['bars'][0]['color']}] "
        f"Watch mode. Ctrl+C to stop.\n"
    )
    incremental = IncrementalIndexer(db)
    watcher = TranscriptWatcher(config.kilo_code_tasks_path, incremental.on_session_changed)
    watcher.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print(f"\n  [{theme['text_muted']}]Watcher stopped.[/{theme['text_muted']}]")
        watcher.stop()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current directory for path matching"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search indexed sessions using hybrid search (semantic + BM25 + recency + path)."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = HybridSearchEngine(db)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Searching for:[/bold] {query}\n")
    
    # Use hybrid search with path matching
    current_dir = str(path) if path else str(Path.cwd())
    results = engine.search(query, current_dir=current_dir, n_results=n_results)
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    if json_output:
        output = [r.to_dict() for r in results]
        console.print(json.dumps(output, indent=2, default=str))
    else:
        # Display results with breakdown
        console.print(f"[dim]Found {len(results)} results (using hybrid search)[/dim]\n")
        
        for i, r in enumerate(results, 1):
            score_pct = f"{r.score:.1%}"
            breakdown = r.breakdown
            
            # Build breakdown string
            breakdown_str = " | ".join([
                f"sem:{breakdown.get('semantic', 0):.2f}",
                f"bm25:{breakdown.get('bm25', 0):.2f}",
                f"rec:{breakdown.get('recency', 0):.2f}",
                f"path:{breakdown.get('path', 0):.2f}"
            ])
            
            # Get technologies if available
            techs = r.metadata.get("technologies", [])
            tech_str = f"\n[dim]Tech: {', '.join(techs[:3])}[/dim]" if techs else ""
            
            # Get files in context
            files = r.metadata.get("files_in_context", [])
            files_str = "\n".join([f"  [dim]* {f}[/dim]" for f in files[:3]]) if files else ""
            
            # Get last active
            last_active = r.metadata.get("last_active", "Unknown")
            if last_active and last_active != "Unknown":
                try:
                    dt = datetime.fromisoformat(last_active)
                    last_active = dt.strftime("%Y-%m-%d")
                except:
                    pass
            
            panel_content = f"[bold]Score:[/bold] {score_pct}\n"
            panel_content += f"[dim]{breakdown_str}[/dim]{tech_str}\n"
            panel_content += f"[dim]Last active: {last_active}[/dim]\n"
            if files_str:
                panel_content += f"\n[bold]Files:[/bold]\n{files_str}"
            
            border = "green" if r.score > 0.7 else "yellow" if r.score > 0.4 else "red"
            
            # Get session title if available
            session_title = r.metadata.get("session_title")
            if session_title:
                title_text = f"[{i}] {session_title}"
            else:
                title_text = f"[{i}] Session {r.session_id[:16]}..."
            
            console.print(Panel(
                panel_content,
                title=title_text,
                border_style=border
            ))
        
        # Show contextual help
        help_manager.show_after_command(
            UserAction.SEARCH,
            db_session_count=db.get_session_count(),
            query=query,
            result_count=len(results)
        )


@app.command()
def detect_fork(
    query: str = typer.Argument(..., help="Describe what you're working on"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of suggestions"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current working directory for path matching"),
    days: int = typer.Option(90, "--days", "-d", help="Limit to sessions from last N days"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Find relevant past sessions to fork context from."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = HybridSearchEngine(db)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold]Detecting fork for:[/bold] {query}",
        title="SmartFork Detect-Fork"
    ))
    
    # Use hybrid search
    current_dir = str(path) if path else str(Path.cwd())
    results = engine.search(query, current_dir=current_dir, n_results=n_results * 2)
    
    # Filter by recency if specified
    if days < 90:
        cutoff = datetime.now() - timedelta(days=days)
        filtered = []
        for r in results:
            last_active = r.metadata.get("last_active")
            if last_active:
                try:
                    dt = datetime.fromisoformat(last_active)
                    if dt > cutoff:
                        filtered.append(r)
                except:
                    filtered.append(r)
            else:
                filtered.append(r)
        results = filtered
    
    if not results:
        console.print("[yellow]No relevant sessions found.[/yellow]")
        return
    
    # Limit results
    results = results[:n_results]
    
    if json_output:
        output = [r.to_dict() for r in results]
        console.print(json.dumps(output, indent=2, default=str))
    else:
        console.print(f"\n[dim]Found {len(results)} relevant session(s):[/dim]\n")
        
        for i, r in enumerate(results, 1):
            score_pct = f"{r.score:.1%}"
            breakdown = r.breakdown
            
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
            
            # Get session title if available
            session_title = r.metadata.get("session_title")
            if session_title:
                title_text = f"[{i}] {session_title}"
            else:
                title_text = f"[{i}] {r.session_id[:20]}..."
            
            # Also show session ID in dim text if title is shown
            id_text = f"\n[dim]ID: {r.session_id[:20]}...[/dim]" if session_title else ""
            
            console.print(Panel(
                f"[bold green]{score_pct}[/bold green] relevance{match_str}{tech_str}{files_preview}\n\n"
                f"[dim]Last active: {last_active}[/dim]{id_text}",
                title=title_text
            ))
        
        console.print("\n[dim]Use [bold]smartfork fork <session_id>[/bold] to generate context file[/dim]")
        
        # Show contextual help
        help_manager.show_after_command(
            UserAction.DETECT_FORK,
            db_session_count=db.get_session_count(),
            query=query,
            result_count=len(results)
        )


@app.command()
def fork(
    session_id: str = typer.Argument(..., help="Session ID to fork context from"),
    query: str = typer.Option("", "--query", "-q", help="Original search query for context"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current working directory"),
):
    """Generate a fork.md context file from a session."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    # Check if session exists
    chunks = db.get_session_chunks(session_id)
    if not chunks:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)
    
    # Get session title if available
    session_title = chunks[0].metadata.session_title if chunks else None
    if session_title:
        console.print(f"[bold]Generating fork.md for session:[/bold] {session_title}")
        console.print(f"[dim]ID: {session_id[:20]}...[/dim]")
    else:
        console.print(f"[bold]Generating fork.md for session:[/bold] {session_id[:20]}...")
    
    # Generate fork.md
    generator = ForkMDGenerator(db)
    current_dir = str(path) if path else str(Path.cwd())
    
    content = generator.generate(session_id, query or "forked session", current_dir)
    
    # Save to file
    if not output:
        short_id = session_id[:8]
        output = Path(f"fork_{short_id}.md")
    
    output.write_text(content, encoding="utf-8")
    
    console.print(f"[green]OK Fork.md saved to:[/green] {output.absolute()}")
    console.print(f"\n[dim]Preview:[/dim]\n{content[:500]}...")
    
    # Show contextual help
    help_manager.show_after_command(
        UserAction.FORK,
        db_session_count=db.get_session_count(),
        session_id=session_id,
        output_file=str(output)
    )


@app.command()
def status():
    """Show indexing status."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    
    # Get stats
    total_chunks = db.get_session_count()
    unique_sessions = len(db.get_unique_sessions())
    
    # Check tasks directory
    if config.kilo_code_tasks_path.exists():
        task_dirs = [d for d in config.kilo_code_tasks_path.iterdir() if d.is_dir()]
        total_tasks = len(task_dirs)
    else:
        total_tasks = 0
    
    # Display status
    console.print(Panel.fit(
        "[bold blue]SmartFork Status[/bold blue]",
        title="Status"
    ))
    
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Kilo Code Tasks Path", str(config.kilo_code_tasks_path))
    table.add_row("Database Path", str(config.chroma_db_path))
    table.add_row("Total Task Directories", str(total_tasks))
    table.add_row("Indexed Sessions", str(unique_sessions))
    table.add_row("Total Chunks", str(total_chunks))
    table.add_row("Index Coverage", f"{unique_sessions}/{total_tasks} ({unique_sessions/max(1,total_tasks)*100:.1f}%)")
    
    console.print(table)
    
    if unique_sessions < total_tasks:
        console.print("\n[yellow]Tip: Run 'smartfork index' to index remaining sessions[/yellow]")
    
    # Show contextual help
    help_manager.show_after_command(
        UserAction.STATUS,
        db_session_count=unique_sessions,
        total_tasks=total_tasks,
        indexed_sessions=unique_sessions
    )


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    console.print(Panel.fit(
        "[bold blue]SmartFork Configuration[/bold blue]",
        title="Config"
    ))

    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for key, value in config.model_dump().items():
        table.add_row(key, str(value))

    console.print(table)


@app.command("config-theme")
def config_theme(
    theme_name: Optional[str] = typer.Argument(None, help="Theme to set"),
    list_all:   bool = typer.Option(False, "--list", "-l", help="List all themes"),
):
    """
    Set or view the color theme.

    Usage:
        smartfork config-theme obsidian    # set theme
        smartfork config-theme --list      # list all themes
        smartfork config-theme             # show current
    """
    from .config import reload_config
    config  = get_config()
    current = getattr(config, "theme", DEFAULT_THEME)

    if list_all or theme_name is None:
        tbl = Table(show_header=True, box=box.SIMPLE)
        tbl.add_column("Theme",  style="bold", width=12)
        tbl.add_column("Description", style="dim", width=40)
        tbl.add_column("",  width=10)
        for tid, td in THEMES.items():
            c0, c1, c2 = [b["color"] for b in td["bars"]]
            swatch = f"[{c0}]▪[/{c0}][{c1}]▪[/{c1}][{c2}]▪[/{c2}] {td['name']}"
            status = "[green]● active[/green]" if tid == current else ""
            tbl.add_row(swatch, td["desc"], status)
        console.print(Panel(tbl, title="[bold]SmartFork Themes[/bold]", box=box.ROUNDED))
        if theme_name is None:
            console.print(f"\n  Current: [bold]{current}[/bold]")
            console.print(f"  Set with: [dim]smartfork config-theme <name>[/dim]\n")
        return

    theme_name = theme_name.lower()
    if theme_name not in THEMES:
        console.print(f"[red]Unknown theme '{theme_name}'[/red]")
        console.print(f"[dim]Valid: {', '.join(THEMES.keys())}[/dim]")
        raise typer.Exit(1)

    # Update and save config
    config.theme = theme_name
    config.save()

    td = THEMES[theme_name]
    c  = td["bars"][1]["color"]
    console.print(f"\n  [{c}]✓[/{c}] Theme → [bold]{td['name']}[/bold] — {td['desc']}")
    console.print(f"  [dim]Saved to {CONFIG_FILE}[/dim]\n")


@app.command()
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Reset the database (WARNING: deletes all indexed data)."""
    if not force:
        confirm = typer.confirm("Are you sure you want to delete all indexed data?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)
    
    config = get_config()
    db = ChromaDatabase(config.chroma_db_path)
    
    db.reset()
    console.print("[green]Database reset complete.[/green]")


@app.command()
def watch():
    """Watch for session changes and index incrementally."""
    config = get_config()
    
    if not config.kilo_code_tasks_path.exists():
        console.print(f"[red]Error: Tasks path does not exist: {config.kilo_code_tasks_path}[/red]")
        raise typer.Exit(1)
    
    console.print("[bold]Starting watcher... Press Ctrl+C to stop.[/bold]\n")
    
    db = ChromaDatabase(config.chroma_db_path)
    incremental = IncrementalIndexer(db)
    watcher = TranscriptWatcher(
        config.kilo_code_tasks_path,
        incremental.on_session_changed
    )
    
    watcher.start()
    
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        watcher.stop()
        console.print("[green]Watcher stopped.[/green]")


# Phase 2: Intelligence Layer Commands

@app.command()
def compaction_check(
    threshold_messages: int = typer.Option(100, "--messages", "-m", help="Message count threshold"),
    threshold_days: int = typer.Option(7, "--days", "-d", help="Age threshold in days"),
):
    """Check for sessions at risk of compaction."""
    from .intelligence.pre_compaction import PreCompactionHook
    
    config = get_config()
    hook = PreCompactionHook(threshold_messages, threshold_days)
    
    with console.status("[bold]Checking sessions..."):
        at_risk = hook.check_sessions(config.kilo_code_tasks_path)
    
    if not at_risk:
        console.print("[green]No sessions at risk of compaction.[/green]")
        return
    
    console.print(Panel.fit(
        f"[bold yellow]{len(at_risk)} sessions at risk[/bold yellow]",
        title="Compaction Check"
    ))
    
    table = Table(show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Messages", style="yellow", justify="right")
    table.add_column("Age (days)", style="blue", justify="right")
    table.add_column("Risk", style="red")
    
    for session in at_risk[:20]:  # Show top 20
        table.add_row(
            session["session_id"][:20],
            str(session["message_count"]),
            str(session["age_days"]),
            session["risk_level"]
        )
    
    console.print(table)
    
    if len(at_risk) > 20:
        console.print(f"\n[dim]... and {len(at_risk) - 20} more[/dim]")


@app.command()
def compaction_export(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be exported"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Export all at-risk sessions"),
):
    """Export sessions before compaction."""
    from .intelligence.pre_compaction import CompactionManager
    
    manager = CompactionManager()
    
    with console.status("[bold]Running compaction export..."):
        results = manager.run_auto_export(dry_run=dry_run)
    
    if dry_run:
        console.print(Panel.fit(
            f"[bold]{results['at_risk']} sessions would be exported[/bold]",
            title="Dry Run"
        ))
    else:
        console.print(Panel.fit(
            f"[bold green]Exported {results['exported']} sessions[/bold green]\n"
            f"Failed: {results['failed']}",
            title="Compaction Export"
        ))
    
    if results['sessions']:
        table = Table(show_header=True)
        table.add_column("Session", style="cyan")
        table.add_column("Action", style="green")
        
        for session in results['sessions'][:10]:
            table.add_row(
                session["session_id"][:20],
                session["action"]
            )
        
        console.print(table)


@app.command()
def cluster_analysis():
    """Analyze session clusters and find duplicates."""
    from .intelligence.clustering import SessionClusterer
    
    clusterer = SessionClusterer()
    
    with console.status("[bold]Analyzing clusters..."):
        analysis = clusterer.analyze_clusters()
    
    console.print(Panel.fit(
        f"[bold]{analysis['total_clusters']} clusters found[/bold]\n"
        f"{analysis['noise_sessions']} unclustered sessions\n"
        f"{analysis['potential_duplicates']} potential duplicates",
        title="Cluster Analysis"
    ))
    
    if analysis['clusters']:
        table = Table(show_header=True)
        table.add_column("Cluster", style="cyan", justify="right")
        table.add_column("Sessions", style="green", justify="right")
        table.add_column("Technologies", style="blue")
        
        for cluster in analysis['clusters'][:10]:
            techs = ", ".join(cluster['common_technologies'][:5])
            table.add_row(
                str(cluster['cluster_id']),
                str(cluster['session_count']),
                techs
            )
        
        console.print("\n[bold]Top Clusters:[/bold]")
        console.print(table)
    
    if analysis['duplicate_pairs']:
        console.print("\n[bold yellow]Potential Duplicates:[/bold yellow]")
        for a, b, sim in analysis['duplicate_pairs'][:5]:
            console.print(f"  • {a[:16]}... ↔ {b[:16]}... ({sim:.1%})")


@app.command()
def tree_build():
    """Build conversation branching tree."""
    from .intelligence.branching import BranchingTree
    
    config = get_config()
    tree = BranchingTree()
    
    with console.status("[bold]Building tree..."):
        tree.auto_build_tree(config.kilo_code_tasks_path)
    
    stats = tree.get_stats()
    
    console.print(Panel.fit(
        f"[bold]{stats['total_sessions']} sessions[/bold] in tree\n"
        f"{stats['root_sessions']} roots, {stats['leaf_sessions']} leaves\n"
        f"Max depth: {stats['max_depth']}",
        title="Tree Built"
    ))


@app.command()
def tree_visualize(
    session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Root session to visualize"),
    expanded: bool = typer.Option(False, "--expanded", "-e", help="Show expanded view with more details"),
):
    """Visualize conversation branching tree."""
    from .intelligence.branching import BranchingTree
    
    tree = BranchingTree()
    stats = tree.get_stats()
    
    # Use compact mode by default for better CLI readability
    tree_text = tree.visualize_tree(session_id, compact=not expanded)
    
    console.print(Panel.fit(
        tree_text,
        title=f"Conversation Tree ({stats['total_sessions']} sessions, {stats['root_sessions']} roots)"
    ))
    
    console.print(f"\n[dim]Stats: {stats['leaf_sessions']} leaves, max depth {stats['max_depth']}[/dim]")
    console.print("[dim]Tip: Use --expanded for more details, or 'smartfork tree-export' for interactive HTML[/dim]")


@app.command()
def tree_export(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(False, "--open", "-b", help="Open in browser after export"),
):
    """Export conversation tree as interactive HTML."""
    from .intelligence.branching import BranchingTree
    
    tree = BranchingTree()
    
    with console.status("[bold]Generating HTML visualization..."):
        html_path = tree.export_html(output)
    
    console.print(f"[green]OK Tree exported to:[/green] {html_path}")
    
    stats = tree.get_stats()
    console.print(Panel.fit(
        f"[bold]{stats['total_sessions']}[/bold] sessions\n"
        f"[bold]{stats['root_sessions']}[/bold] root sessions\n"
        f"[bold]{stats['leaf_sessions']}[/bold] leaf sessions\n"
        f"Max depth: [bold]{stats['max_depth']}[/bold]",
        title="Tree Statistics"
    ))
    
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
        console.print("[dim]Opened in browser[/dim]")


@app.command()
def vault_add(
    session_id: str = typer.Argument(..., help="Session ID to add to vault"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Encryption password"),
):
    """Add a session to the privacy vault."""
    from .intelligence.privacy import PrivacyVault
    
    config = get_config()
    
    if not password:
        password = typer.prompt("Enter vault password", hide_input=True)
    
    vault = PrivacyVault(password)
    
    # Find session directory
    task_dir = config.kilo_code_tasks_path / session_id
    if not task_dir.exists():
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)
    
    with console.status("[bold]Adding to vault..."):
        success = vault.add_to_vault(session_id, task_dir)
    
    if success:
        console.print(f"[green]OK Session {session_id[:16]}... added to vault[/green]")
    else:
        console.print("[red]Failed to add to vault[/red]")


@app.command()
def vault_list():
    """List vaulted sessions."""
    from .intelligence.privacy import PrivacyVault
    
    vault = PrivacyVault()
    sessions = vault.list_vaulted_sessions()
    
    if not sessions:
        console.print("[dim]No sessions in vault[/dim]")
        return
    
    console.print(Panel.fit(
        f"[bold]{len(sessions)} vaulted sessions[/bold]",
        title="Privacy Vault"
    ))
    
    table = Table(show_header=True)
    table.add_column("Session", style="cyan")
    table.add_column("Vaulted At", style="green")
    table.add_column("Files", style="blue", justify="right")
    
    for session in sessions:
        table.add_row(
            session["session_id"][:20],
            session.get("vaulted_at", "unknown"),
            str(session.get("file_count", 0))
        )
    
    console.print(table)


@app.command()
def vault_restore(
    session_id: str = typer.Argument(..., help="Session ID to restore"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Decryption password"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Restore a session from the vault."""
    from .intelligence.privacy import PrivacyVault
    
    if not password:
        password = typer.prompt("Enter vault password", hide_input=True)
    
    vault = PrivacyVault(password)
    
    with console.status("[bold]Restoring from vault..."):
        result = vault.restore_from_vault(session_id, output)
    
    if result:
        console.print(f"[green]OK Session restored to: {result}[/green]")
    else:
        console.print("[red]Failed to restore session[/red]")


@app.command()
def vault_search(
    query: str = typer.Argument(..., help="Search query"),
    password: Optional[str] = typer.Option(None, "--password", "-p", help="Decryption password"),
):
    """Search within vaulted sessions."""
    from .intelligence.privacy import PrivacyVault
    
    if not password:
        password = typer.prompt("Enter vault password", hide_input=True)
    
    vault = PrivacyVault(password)
    
    with console.status("[bold]Searching vault..."):
        results = vault.search_vault(query)
    
    console.print(Panel.fit(
        f"[bold]{len(results)} results[/bold]",
        title="Vault Search"
    ))
    
    for r in results[:10]:
        console.print(Panel(
            f"[dim]{r['preview']}[/dim]",
            title=f"{r['session_id'][:16]}... / {r['file']}"
        ))


# Phase 3: Testing and Metrics Commands

@app.command()
def test(
    suite: Optional[str] = typer.Option(None, "--suite", "-s", help="Test suite to run (indexer, search, database, fork)"),
):
    """Run SmartFork tests."""
    from .testing.test_runner import create_default_test_runner
    
    runner = create_default_test_runner()
    
    if suite:
        with console.status(f"[bold]Running {suite} tests..."):
            result = runner.run_suite(suite)
        suites = [result]
    else:
        with console.status("[bold]Running all tests..."):
            suites = runner.run_all()
    
    # Display results
    for suite in suites:
        color = "green" if suite.failed_count == 0 else "red"
        console.print(Panel.fit(
            f"[bold {color}]{suite.passed_count}/{len(suite.tests)} passed[/bold {color}]\n"
            f"Duration: {suite.total_duration_ms:.0f}ms",
            title=f"Test Suite: {suite.name}"
        ))
        
        if suite.failed_count > 0:
            table = Table(show_header=True)
            table.add_column("Test", style="cyan")
            table.add_column("Status", style="red")
            table.add_column("Error", style="dim")
            
            for test in suite.tests:
                if not test.passed:
                    table.add_row(
                        test.name,
                        "FAILED",
                        test.error_message[:50] + "..." if test.error_message and len(test.error_message) > 50 else (test.error_message or "")
                    )
            
            console.print(table)
    
    summary = runner.get_summary()
    console.print(f"\n[dim]Total: {summary['passed']}/{summary['total_tests']} passed "
                  f"({summary['pass_rate']:.1%})[/dim]")


@app.command()
def metrics(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to show"),
):
    """Show success metrics dashboard."""
    from .testing.metrics_tracker import MetricsTracker
    
    tracker = MetricsTracker()
    data = tracker.get_dashboard_data(days)
    
    console.print(Panel.fit(
        f"[bold]Success Metrics[/bold] (last {data['period_days']} days)",
        title="Metrics Dashboard"
    ))
    
    # Key metrics
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    km = data['key_metrics']
    table.add_row("Unique Sessions", str(data['unique_sessions']))
    table.add_row("Avg Fork Gen Time", f"{km['avg_fork_generation_time_ms']:.0f}ms")
    table.add_row("Context Recovered", f"{km['total_context_recovered_mb']:.1f}MB")
    table.add_row("Sessions/Day", f"{km['sessions_per_day']:.1f}")
    
    console.print(table)
    
    # Metric summaries
    if data['metric_summaries']:
        console.print("\n[bold]Metric Trends:[/bold]")
        for name, summary in data['metric_summaries'].items():
            trend_color = {
                'improving': 'green',
                'stable': 'yellow',
                'degrading': 'red',
                'insufficient_data': 'dim'
            }.get(summary['trend'], 'white')
            
            console.print(f"  {name}: {summary['mean']:.2f} "
                          f"([{trend_color}]{summary['trend']}[/{trend_color}])")


@app.command()
def ab_test_status():
    """Show A/B test status."""
    from .testing.ab_testing import ABTestManager
    
    manager = ABTestManager()
    summary = manager.get_test_summary()
    
    console.print(Panel.fit(
        f"[bold]{summary['total_tests']}[/bold] active tests\n"
        f"[bold]{summary['total_sessions']}[/bold] test sessions",
        title="A/B Testing"
    ))
    
    if summary['active_tests']:
        table = Table(show_header=True)
        table.add_column("Test", style="cyan")
        table.add_column("Sessions", style="green", justify="right")
        table.add_column("Control", style="blue", justify="right")
        table.add_column("Treatment", style="yellow", justify="right")
        table.add_column("Result", style="magenta")
        
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
        
        console.print(table)


@app.command()
def update_titles(
    force: bool = typer.Option(False, "--force", "-f", help="Force regeneration of all titles"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Show what would be generated without updating"),
):
    """Generate or update session titles for all indexed sessions."""
    config = get_config()
    
    db = ChromaDatabase(config.chroma_db_path)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    # Get all unique sessions
    session_ids = db.get_unique_sessions()
    
    if not session_ids:
        console.print("[yellow]No sessions found in database.[/yellow]")
        raise typer.Exit(0)
    
    console.print(Panel.fit(
        f"[bold blue]Update Session Titles[/bold blue]\n"
        f"Found {len(session_ids)} sessions to process",
        title="SmartFork"
    ))
    
    # Initialize title generator
    title_gen = TitleGenerator()
    title_manager = TitleManager(db, title_gen)
    parser = KiloCodeParser()
    
    # Track results
    updated = 0
    skipped = 0
    failed = 0
    
    # Process each session
    with console.status("[bold]Generating titles...") as status:
        for i, session_id in enumerate(session_ids):
            try:
                # Check if session already has a title (unless force)
                if not force:
                    chunks = db.get_session_chunks(session_id)
                    if chunks and chunks[0].metadata.session_title:
                        skipped += 1
                        continue
                
                # Parse the session to get full content
                task_dir = config.kilo_code_tasks_path / session_id
                if not task_dir.exists():
                    failed += 1
                    logger.warning(f"Session directory not found: {task_dir}")
                    continue
                
                session = parser.parse_task_directory(task_dir)
                if not session:
                    failed += 1
                    logger.warning(f"Could not parse session: {session_id}")
                    continue
                
                # Generate title
                title = title_manager.generate_and_store_title(session)
                
                if dry_run:
                    console.print(f"[dim]{session_id[:16]}...[/dim] -> {title}")
                else:
                    # Re-index the session to store the new title
                    # Note: This requires re-indexing since ChromaDB doesn't support metadata updates
                    indexer = FullIndexer(db, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
                    indexer.index_session(task_dir)
                    
                updated += 1
                
                if (i + 1) % 10 == 0:
                    status.update(f"[bold]Processed {i + 1}/{len(session_ids)} sessions...")
                    
            except Exception as e:
                logger.error(f"Failed to update title for {session_id}: {e}")
                failed += 1
    
    # Display results
    console.print("\n[bold]Results:[/bold]")
    console.print(f"  [green]Updated:[/green] {updated}")
    console.print(f"  [yellow]Skipped:[/yellow] {skipped}")
    if failed > 0:
        console.print(f"  [red]Failed:[/red] {failed}")
    
    if dry_run:
        console.print("\n[dim]This was a dry run. Use without --dry-run to apply changes.[/dim]")
    else:
        console.print("\n[green]Title update complete![/green]")


@app.command()
def interactive():
    """Start the interactive shell (REPL mode)."""
    console.print(Panel.fit(
        "[bold cyan]SmartFork Interactive Mode[/bold cyan]\n"
        "Starting interactive shell...",
        title="SmartFork"
    ))
    
    try:
        start_interactive_shell()
    except Exception as e:
        console.print(f"[red]Error starting interactive shell: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
